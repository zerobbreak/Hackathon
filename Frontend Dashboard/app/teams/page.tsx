"use client"

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Plus, Search, Users, User, MapPin, Phone, Mail, Edit, Trash2, Activity, CheckCircle, XCircle, Clock, Calendar } from 'lucide-react'
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

export default function TeamsPage() {
  const initialTeams = [
    {
      id: "T001",
      name: "Search & Rescue Alpha",
      members: [
        { name: "John Doe", role: "Leader", status: "Available" },
        { name: "Alice Brown", role: "Medic", status: "Available" },
        { name: "Bob Green", role: "Specialist", status: "Deployed" },
      ],
      status: "Deployed",
      currentAssignment: "Riverbend",
      contact: "John Doe",
      lastActivity: "5 mins ago",
    },
    {
      id: "T002",
      name: "Medical Response Bravo",
      members: [
        { name: "Jane Smith", role: "Leader", status: "Available" },
        { name: "Chris White", role: "Paramedic", status: "Available" },
      ],
      status: "Standby",
      currentAssignment: "N/A",
      contact: "Jane Smith",
      lastActivity: "2 hours ago",
    },
    {
      id: "T003",
      name: "Logistics Charlie",
      members: [
        { name: "Mike Johnson", role: "Leader", status: "Deployed" },
        { name: "Emily Davis", role: "Coordinator", status: "Available" },
        { name: "David Lee", role: "Driver", status: "Deployed" },
      ],
      status: "Deployed",
      currentAssignment: "Supply Hub",
      contact: "Mike Johnson",
      lastActivity: "1 hour ago",
    },
    {
      id: "T004",
      name: "Communication Delta",
      members: [
        { name: "Sarah Lee", role: "Leader", status: "Available" },
        { name: "Tom Wilson", role: "Operator", status: "Available" },
      ],
      status: "Active",
      currentAssignment: "Coastal Area A",
      contact: "Sarah Lee",
      lastActivity: "30 mins ago",
    },
  ];

  const [teams, setTeams] = useState(initialTeams);
  const [searchTerm, setSearchTerm] = useState('');

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "Deployed":
        return <Badge className="bg-red-500 hover:bg-red-600 text-white">{status}</Badge>
      case "Active":
        return <Badge className="bg-green-500 hover:bg-green-600 text-white">{status}</Badge>
      case "Standby":
        return <Badge variant="secondary">{status}</Badge>
      default:
        return <Badge>{status}</Badge>
    }
  }

  const getMemberStatusIcon = (status: string) => {
    switch (status) {
      case "Available":
        return <CheckCircle className="h-3 w-3 text-green-500" />
      case "Deployed":
        return <Clock className="h-3 w-3 text-yellow-500" />
      case "Unavailable":
        return <XCircle className="h-3 w-3 text-red-500" />
      default:
        return null
    }
  }

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    console.log("Searching for:", e.target.value);
  };

  const handleNewTeam = () => {
    alert("Opening form to create a new team!");
  };

  const handleContactLeader = (teamId: string, contactPerson: string) => {
    alert(`Contacting leader ${contactPerson} for team ${teamId}`);
  };

  const handleEmailTeam = (teamId: string) => {
    alert(`Sending email to team ${teamId}`);
  };

  const handleViewSchedule = (teamId: string) => {
    alert(`Viewing schedule for team ${teamId}`);
  };

  const handleEditTeam = (teamId: string) => {
    alert(`Editing team: ${teamId}`);
  };

  const handleDisbandTeam = (teamId: string) => {
    if (confirm(`Are you sure you want to disband team ${teamId}?`)) {
      setTeams(teams.filter(team => team.id !== teamId));
      alert(`Team ${teamId} disbanded.`);
    }
  };

  const filteredTeams = teams.filter(team =>
    team.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    team.contact.toLowerCase().includes(searchTerm.toLowerCase()) ||
    team.members.some(member => member.name.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle>Response Teams</CardTitle>
        <CardDescription>Manage and coordinate emergency response teams and their members.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search teams or members..."
              className="w-full rounded-lg bg-background pl-8"
              value={searchTerm}
              onChange={handleSearchChange}
            />
          </div>
          <Button size="sm" className="h-8 gap-1" onClick={handleNewTeam}>
            <Plus className="h-3.5 w-3.5" />
            <span className="sr-only sm:not-sr-only sm:whitespace-nowrap">New Team</span>
          </Button>
        </div>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Team Name</TableHead>
              <TableHead>Members</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Current Assignment</TableHead>
              <TableHead>Last Activity</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredTeams.map((team) => (
              <TableRow key={team.id}>
                <TableCell className="font-medium flex items-center gap-2">
                  <Users className="h-4 w-4 text-muted-foreground" /> {team.name}
                </TableCell>
                <TableCell>
                  <div className="flex items-center -space-x-2">
                    {team.members.slice(0, 3).map((member, index) => (
                      <Avatar key={index} className="h-7 w-7 border-2 border-background">
                        <AvatarImage src={`/placeholder.svg?height=28&width=28&query=${member.name}`} />
                        <AvatarFallback>{member.name.split(' ').map(n => n[0]).join('')}</AvatarFallback>
                      </Avatar>
                    ))}
                    {team.members.length > 3 && (
                      <Avatar className="h-7 w-7 border-2 border-background">
                        <AvatarFallback>+{team.members.length - 3}</AvatarFallback>
                      </Avatar>
                    )}
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="link" size="sm" className="h-auto p-0 text-xs text-muted-foreground">
                        View all ({team.members.length})
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      {team.members.map((member, index) => (
                        <DropdownMenuItem key={index} className="flex items-center gap-2">
                          {getMemberStatusIcon(member.status)} {member.name} ({member.role})
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
                <TableCell>{getStatusBadge(team.status)}</TableCell>
                <TableCell className="flex items-center gap-1">
                  {team.currentAssignment !== "N/A" && <MapPin className="h-4 w-4 text-muted-foreground" />}
                  {team.currentAssignment}
                </TableCell>
                <TableCell className="flex items-center gap-1">
                  <Clock className="h-3 w-3 text-muted-foreground" /> {team.lastActivity}
                </TableCell>
                <TableCell className="text-right">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <span className="sr-only">Actions</span>
                        <Activity className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => handleContactLeader(team.id, team.contact)}>
                        <Phone className="mr-2 h-4 w-4" /> Contact Leader
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleEmailTeam(team.id)}>
                        <Mail className="mr-2 h-4 w-4" /> Email Team
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleViewSchedule(team.id)}>
                        <Calendar className="mr-2 h-4 w-4" /> View Schedule
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => handleEditTeam(team.id)}>
                        <Edit className="mr-2 h-4 w-4" /> Edit Team
                      </DropdownMenuItem>
                      <DropdownMenuItem className="text-red-600" onClick={() => handleDisbandTeam(team.id)}>
                        <Trash2 className="mr-2 h-4 w-4" /> Disband Team
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}
